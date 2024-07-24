import NewDocumentLink from '@/components/new-document-link';
import { useTranslate } from '@/hooks/common-hooks';
import { ITestingDocument } from '@/interfaces/database/knowledge';
import { EyeOutlined } from '@ant-design/icons';
import { Button, Table, TableProps, Tooltip } from 'antd';
import { useDispatch, useSelector } from 'umi';

interface IProps {
  handleTesting: () => Promise<any>;
}

const SelectFiles = ({ handleTesting }: IProps) => {
  const documents: ITestingDocument[] = useSelector(
    (state: any) => state.testingModel.documents,
  );
  const { t } = useTranslate('fileManager');

  const dispatch = useDispatch();

  const columns: TableProps<ITestingDocument>['columns'] = [
    {
      title: 'Name',
      dataIndex: 'doc_name',
      key: 'doc_name',
      render: (text) => <p>{text}</p>,
    },

    {
      title: 'Hits',
      dataIndex: 'count',
      key: 'count',
      width: 80,
    },
    {
      title: 'View',
      key: 'view',
      width: 50,
      render: (_, { doc_id, doc_name }) => (
        <NewDocumentLink
          documentName={doc_name}
          documentId={doc_id}
          prefix="document"
        >
          <Tooltip title={t('preview')}>
            <Button type="text">
              <EyeOutlined size={20} />
            </Button>
          </Tooltip>
        </NewDocumentLink>
      ),
    },
  ];

  const rowSelection = {
    onChange: (selectedRowKeys: React.Key[]) => {
      dispatch({
        type: 'testingModel/setSelectedDocumentIds',
        payload: selectedRowKeys,
      });
      handleTesting();
    },
    getCheckboxProps: (record: ITestingDocument) => ({
      disabled: record.doc_name === 'Disabled User', // Column configuration not to be checked
      name: record.doc_name,
    }),
  };

  return (
    <Table
      columns={columns}
      dataSource={documents}
      showHeader={false}
      rowSelection={rowSelection}
      rowKey={'doc_id'}
    />
  );
};

export default SelectFiles;
