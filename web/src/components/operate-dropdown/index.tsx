import { useShowDeleteConfirm } from '@/hooks/commonHooks';
import { DeleteOutlined, MoreOutlined } from '@ant-design/icons';
import { Dropdown, MenuProps, Space } from 'antd';
import { useTranslation } from 'react-i18next';

import React, { useMemo } from 'react';
import styles from './index.less';

interface IProps {
  deleteItem: () => Promise<any> | void;
  iconFontSize?: number;
  items?: MenuProps['items'];
}

const OperateDropdown = ({
  deleteItem,
  children,
  iconFontSize = 30,
  items: otherItems = [],
}: React.PropsWithChildren<IProps>) => {
  const { t } = useTranslation();
  const showDeleteConfirm = useShowDeleteConfirm();

  const handleDelete = () => {
    showDeleteConfirm({ onOk: deleteItem });
  };

  const handleDropdownMenuClick: MenuProps['onClick'] = ({ domEvent, key }) => {
    domEvent.preventDefault();
    domEvent.stopPropagation();
    if (key === '1') {
      handleDelete();
    }
  };

  const items: MenuProps['items'] = useMemo(() => {
    return [
      {
        key: '1',
        label: (
          <Space>
            {t('common.delete')}
            <DeleteOutlined />
          </Space>
        ),
      },
      ...otherItems,
    ];
  }, [t, otherItems]);

  return (
    <Dropdown
      menu={{
        items,
        onClick: handleDropdownMenuClick,
      }}
    >
      {children || (
        <span className={styles.delete}>
          <MoreOutlined
            rotate={90}
            style={{ fontSize: iconFontSize, color: 'gray', cursor: 'pointer' }}
          />
        </span>
      )}
    </Dropdown>
  );
};

export default OperateDropdown;
