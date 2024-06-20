import { IModalManagerChildrenProps } from '@/components/modal-manager';
import { useTranslate } from '@/hooks/commonHooks';
import { Form, Input, Modal } from 'antd';

interface IProps extends Omit<IModalManagerChildrenProps, 'showModal'> {
  loading: boolean;
  onOk: (name: string) => void;
}

const FolderCreateModal = ({ visible, hideModal, loading, onOk }: IProps) => {
  const [form] = Form.useForm();
  const { t } = useTranslate('common');

  type FieldType = {
    name?: string;
  };

  const handleOk = async () => {
    const ret = await form.validateFields();

    return onOk(ret.name);
  };

  const handleCancel = () => {
    hideModal();
  };

  const onFinish = (values: any) => {
    console.log('Success:', values);
  };

  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
  };

  return (
    <Modal
      title={t('newFolder', { keyPrefix: 'fileManager' })}
      open={visible}
      onOk={handleOk}
      onCancel={handleCancel}
      okButtonProps={{ loading }}
      confirmLoading={loading}
    >
      <Form
        name="basic"
        labelCol={{ span: 4 }}
        wrapperCol={{ span: 20 }}
        style={{ maxWidth: 600 }}
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
        autoComplete="off"
        form={form}
      >
        <Form.Item<FieldType>
          label={t('name')}
          name="name"
          rules={[{ required: true, message: t('namePlaceholder') }]}
        >
          <Input />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default FolderCreateModal;
